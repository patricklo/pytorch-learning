package com.example.repository;

import com.example.entity.ClientOrder;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Lock;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import jakarta.persistence.LockModeType;
import java.util.List;
import java.util.Optional;

@Repository
public interface ClientOrderRepository extends JpaRepository<ClientOrder, Long> {

    Optional<ClientOrder> findByOrderNumber(String orderNumber);

    @Lock(LockModeType.PESSIMISTIC_WRITE)
    @Query("SELECT co FROM ClientOrder co WHERE co.id = :id")
    Optional<ClientOrder> findByIdWithPessimisticLock(@Param("id") Long id);

    @Lock(LockModeType.OPTIMISTIC)
    @Query("SELECT co FROM ClientOrder co WHERE co.id = :id")
    Optional<ClientOrder> findByIdWithOptimisticLock(@Param("id") Long id);

    List<ClientOrder> findByClientId(String clientId);

    List<ClientOrder> findByStatus(String status);

    @Query("SELECT co FROM ClientOrder co WHERE co.clientId = :clientId AND co.status = :status")
    List<ClientOrder> findByClientIdAndStatus(@Param("clientId") String clientId, @Param("status") String status);

    @Query("SELECT COUNT(co) > 0 FROM ClientOrder co WHERE co.orderNumber = :orderNumber")
    boolean existsByOrderNumber(@Param("orderNumber") String orderNumber);
} 